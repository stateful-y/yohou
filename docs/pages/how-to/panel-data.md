# How to Work with Panel Data

This guide shows you how to forecast multiple related time series (stores,
sensors, regions) in a single call using Yohou's panel data support. Panel
DataFrames use the `{group}__{column}` naming convention, and every group
must share the same column suffixes.

## Prerequisites

- Yohou installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with the fit/predict lifecycle ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Structure Your Data as a Panel

Name columns with `{group}__{variable}` so Yohou detects panel structure
automatically. Columns without a `__` prefix are treated as global (shared
across all groups):

```python
import polars as pl

y = pl.DataFrame({
    "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
    "store_a__sales": [100, 110, 120, 130, 140, 150],
    "store_b__sales": [200, 210, 220, 230, 240, 250],
})
```

To verify that Yohou parses your columns correctly, use
[`inspect_panel`](/pages/api/generated/yohou.utils.inspect_panel/):

```python
from yohou.utils.panel import inspect_panel

global_names, panel_groups = inspect_panel(y)
print(global_names)    # []
print(panel_groups)    # {'store_a': ['store_a__sales'], 'store_b': ['store_b__sales']}
```

If you already have per-group DataFrames,
[`dict_to_panel`](/pages/api/generated/yohou.utils.dict_to_panel/)
joins them into the expected format:

```python
from yohou.utils.panel import dict_to_panel

y = dict_to_panel({
    "store_a": df_store_a,  # DataFrame with "time" and "sales" columns
    "store_b": df_store_b,
})
# Result has columns: "time", "store_a__sales", "store_b__sales"
```

To extract a single group back out, use
[`get_group_df`](/pages/api/generated/yohou.utils.get_group_df/).

## 2. Fit a Forecaster

Pass panel DataFrames directly to `fit`. Use [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/) to create train/test sets.
The default `panel_strategy="global"` gives each group independent transformers while
sharing a single [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) model across all groups:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.model_selection import train_test_split

y_train, y_test = train_test_split(y, test_size=2)

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    panel_strategy="global",
)
forecaster.fit(y_train, forecasting_horizon=2)
y_pred = forecaster.predict()
```

## 3. Predict and Observe for Specific Groups

Use `groups` to predict or observe only a subset of panel groups. This is
useful when new data arrives for some entities but not others:

```python
y_pred_a = forecaster.predict(groups=["store_a"])
```

The same parameter works on `observe` and `rewind`, so you can update one
group's observation window without touching the others:

```python
forecaster.observe(y_new, groups=["store_a"])
```

## 4. Choose a Panel Strategy

The `panel_strategy` parameter controls how groups share information:

- **`"global"`** (default): each group gets independent transformers, but
  all groups contribute rows to a single model. Best when groups share
  similar dynamics.
- **`"multivariate"`**: the entire panel is treated as one wide multivariate
  series. Use this for small panels (fewer than ~20 groups) where
  cross-group correlations are strong.

For completely independent models per group, use
[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/)
instead. It clones the forecaster (for example, [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/))
and fits one instance per group, which is
best when groups have genuinely different dynamics and enough history each:

```python
from yohou.compose import LocalPanelForecaster
from yohou.point import SeasonalNaive

local = LocalPanelForecaster(
    forecaster=SeasonalNaive(seasonality=12),
)
local.fit(y_train, forecasting_horizon=2)
```

See [Panel Data](../explanation/panel-data.md) for the full explanation of
each strategy and when to pick one over another.

## 5. Score Panel Forecasts

Scorers handle panel data automatically. Use `aggregation_method="groupwise"`
to get one score per group so you can spot underperforming entities:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError(aggregation_method="groupwise")
scorer.fit(y_train)
scores = scorer.score(y_test, y_pred)  # one row per group
```

To weight groups differently in the aggregated score, pass a dict mapping
group names to weights:

```python
scorer = MeanAbsoluteError(
    aggregation_method="all",
    groups={"store_a": 2.0, "store_b": 1.0},
)
scorer.fit(y_train)
scalar_score = scorer.score(y_test, y_pred)  # store_a has twice the influence
```

See [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for the
complete scoring workflow and
[Forecast Accuracy](../explanation/forecast-accuracy.md) for aggregation
mode details.

## 6. Add Exogenous Features

Exogenous features can include both global columns (shared across groups)
and local columns (group-specific). Global columns lack the `__` prefix:

```python
X_actual = pl.DataFrame({
    "time": [...],
    "holiday": [True, False, ...],              # global, shared
    "store_a__promotion": [0.1, 0.2, ...],      # local to store_a
    "store_b__promotion": [0.0, 0.1, ...],      # local to store_b
})
```

Pass panel exogenous data to `fit()` as `X_actual=X_actual`. For
known-future features (e.g., holidays), use `X_future`. See
[Use Exogenous Features](exogenous-features.md) for the full guide.

!!! tip
    Ensemble forecasters
    ([`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.VotingPointForecaster/),
    [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.VotingIntervalForecaster/),
    [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.VotingClassProbaForecaster/))
    support panel data automatically, with aggregation per group. See
    [Ensemble Forecasting](ensemble-forecasting.md) for the full workflow.

## See Also

- [Panel Data Tutorial](../tutorials/panel-data.md): hands-on introduction to panel forecasting
- [Panel Data](../explanation/panel-data.md): the panel data model, naming convention rationale, and strategy trade-offs
- [Use Exogenous Features](exogenous-features.md): global and local exogenous columns in panel context
- [Visualize Forecasts](visualize-forecasts.md): automatic panel faceting for forecast and residual plots
- [API Reference: yohou.utils.panel](../api/utils.md)
