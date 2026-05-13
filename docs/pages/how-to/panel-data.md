# How to Work with Panel Data

Panel data (multiple related time series) uses a column naming convention with double underscores: `{group}__{column}`. Every group must share the same column suffixes.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

```python
import polars as pl

y = pl.DataFrame({
    "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
    "store_a__sales": [100, 110, 120, 130, 140, 150],
    "store_b__sales": [200, 210, 220, 230, 240, 250],
})
```

## Inspect Panel Structure

[`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) separates global (unprefixed) columns from panel groups:

```python
from yohou.utils.panel import inspect_panel

global_names, panel_groups = inspect_panel(y)
print(global_names)    # []
print(panel_groups)    # {'store_a': ['store_a__sales'], 'store_b': ['store_b__sales']}
```

## Extract a Single Group

[`get_group_df`](/pages/api/generated/yohou.utils.panel.get_group_df/) extracts one group's data with unprefixed column names:

```python
from yohou.utils.panel import get_group_df

schema = {"sales": pl.Float64}
df_a = get_group_df(y, "store_a", schema)
print(df_a.columns)  # ['time', 'sales']
```

## Convert a Dict Back to Panel Format

[`dict_to_panel`](/pages/api/generated/yohou.utils.panel.dict_to_panel/) joins per-group DataFrames back into a single wide DataFrame:

```python
from yohou.utils.panel import dict_to_panel

data = {
    "store_a": pl.DataFrame({"time": [...], "sales": [100, 110]}),
    "store_b": pl.DataFrame({"time": [...], "sales": [200, 210]}),
}
panel_df = dict_to_panel(data)
# columns: ["time", "store_a__sales", "store_b__sales"]
```

## Fit a Forecaster on Panel Data

Pass panel DataFrames directly to `fit`. With `panel_strategy="global"` (the default), each group gets independent transformers but shares a single model:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    panel_strategy="global",
)
forecaster.fit(y_train, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Predict for Specific Groups

Use `groups` to observe or predict only a subset of groups:

```python
y_pred_a = forecaster.predict(
    forecasting_horizon=12,
    groups=["store_a"],
)
```

To assign different weights to groups during scoring, pass a dict instead of a
list:

```python
scorer = PointScorer(groups={"store_a": 2.0, "store_b": 1.0})
```

This scores both groups but gives `store_a` twice the influence in the
aggregated result.

## Global vs Multivariate Strategy

- **`"global"`** (default): Detects panel groups, fits separate transformers per group, pools data for the estimator. Best when groups share similar dynamics.
- **`"multivariate"`**: Treats `__` prefixed columns as ordinary multivariate features. One transformer and model see the full wide DataFrame. Best for modeling cross-group relationships.

```python
# Treat all columns as a single multivariate series
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    panel_strategy="multivariate",
)
```

## Exogenous Features with Panel Data

Exogenous features (`X_actual`) can include both global columns (shared across groups) and local columns (group-specific). Global columns lack the `__` prefix:

```python
X_actual = pl.DataFrame({
    "time": [...],
    "holiday": [True, False, ...],              # global - shared
    "store_a__promotion": [0.1, 0.2, ...],      # local to store_a
    "store_b__promotion": [0.0, 0.1, ...],      # local to store_b
})
```

Pass panel exogenous data to `fit()` as `X_actual=X_actual`. For known-future features (e.g., holidays for all groups), use `X_future`. For external forecast vintages, use `X_forecast`.

For common panel data errors and their fixes, see [Troubleshooting](troubleshooting.md#panel-column-naming-errors).

Ensemble forecasters ([`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/), [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/),
[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/)) support panel data automatically. Each base
forecaster receives the full panel, and aggregation happens per-group.
