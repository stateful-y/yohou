# Troubleshooting

## Missing or Invalid "time" Column

**Error**: `"must contain a 'time' column"` or `"'time' column must have dtype pl.Datetime or pl.Date"`

Every DataFrame passed to yohou must have a `"time"` column with a datetime type:

```python
import polars as pl

# Wrong: string dates
df = pl.DataFrame({"time": ["2020-01-01", "2020-02-01"], "value": [1.0, 2.0]})

# Fix: cast to datetime
df = df.with_columns(pl.col("time").str.to_datetime())
```

**Error**: `"'time' column must be sorted in ascending order"`

Sort before passing to any yohou function:

```python
df = df.sort("time")
```

## Not Enough Data for Observation Horizon

**Error**: `"Not enough data to set observed y"`

The forecaster or transformer needs at least `observation_horizon` rows of historical data. Check your data length exceeds the required horizon:

```python
print(f"Data rows: {len(y_train)}")
print(f"Observation horizon: {forecaster.observation_horizon}")
```

The observation horizon is the maximum of the forecaster's own horizon and any transformer horizons (e.g., `SeasonalDifferencing(seasonality=132)` requires at least 132 rows).

## Panel Column Naming Errors

**Error**: `"Panel column names conflict with global column names"`

A column like `x__a` creates ambiguity if a global column named `a` also exists. Rename one of them:

```python
# Wrong: both "a" and "x__a" exist
df = pl.DataFrame({"time": [...], "a": [...], "x__a": [...]})

# Fix: rename the global column
df = df.rename({"a": "global_a"})
```

**Error**: `"The local groups in y do not have the same column suffixes"`

All panel groups must have identical column structures. If group `store_a` has `sales` and `returns`, group `store_b` needs both too:

```python
# Wrong: store_a has 2 columns, store_b has 1
y = pl.DataFrame({
    "time": [...],
    "store_a__sales": [...],
    "store_a__returns": [...],
    "store_b__sales": [...],  # missing store_b__returns
})
```

**Error**: `"y contains both local and standard columns"`

Target `y` cannot mix unprefixed and `__` prefixed columns. Either use all global columns or all panel columns.

## Interval Coverage Rate Errors

**Error**: `"All coverage_rates must be in (0, 1]"`

Coverage rates are proportions, not percentages:

```python
# Wrong: using percentages
y_pred_interval = forecaster.predict_interval(coverage_rates=[90, 95])

# Fix: use proportions
y_pred_interval = forecaster.predict_interval(coverage_rates=[0.9, 0.95])
```

## Time Weight Errors

**Error**: `"time_weight callable must accept either 1 parameter or 2 parameters"`

Weight functions must accept `(time: pl.Series)` or `(time: pl.Series, group_name: str)`:

```python
# Wrong: no parameters
weight_fn = lambda: 1.0

# Fix: accept time series
from yohou.utils.weighting import exponential_decay_weight
weight_fn = exponential_decay_weight(half_life=365)
```

**Error**: `"time_weight DataFrame must have 'weight' column"`

When passing a DataFrame as `time_weight`, it must have `"time"` and `"weight"` columns:

```python
weights = pl.DataFrame({
    "time": y_train["time"],
    "weight": [1.0] * len(y_train),
})
```

## Exogenous Feature Errors

**Error**: `"target_as_feature=None requires X to be provided"`

If the forecaster uses exogenous features (`ignores_exogenous=False`), you must pass `X_actual` at fit time. At predict time, use `X_future` for known-ahead features or `X_forecast` for external forecast vintages:

```python
forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=12)
y_pred = forecaster.predict(X_future=X_test, forecasting_horizon=12)
```

## Build Errors After Moving Documentation

If documentation builds fail after restructuring, check for stale cross-links:

```bash
grep -rn "getting-started\|user-guide\|development/" docs/pages/ --include="*.md"
```

Fix any matches to use the new paths under `tutorials/`, `how-to/`, `explanation/`, or `reference/`.

## Plotting ImportError

If you see `ImportError: plotly is not installed`, install the plotting extra:

```bash
pip install yohou[plotting]
```

## Class-Probability Forecasting Errors

**Numeric data passed to `ClassProbaReductionForecaster`**: The target column must
contain string or categorical values, not floats. Convert with
`y.with_columns(pl.col("target").cast(pl.Utf8))`.

**Mismatched class labels between train and test**: Ensure the test data contains
only classes that appeared during training. New classes at prediction time will
cause errors.

## Ensemble Errors

**Base forecasters must support the same prediction type**: All forecasters passed
to [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) must be point forecasters.
[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) requires interval forecasters, and
[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) requires class-probability forecasters.
