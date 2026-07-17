# Panel Data

In this tutorial, we will forecast multiple related time series simultaneously using panel data. Many real forecasting tasks involve groups of related series: regional sales, sensor networks, tourism by destination. Yohou represents these as a single DataFrame where column names encode the group with a `__` separator (e.g. `T187__tourists`, `T188__tourists`). We will load a multi-series tourism dataset, inspect its panel structure, fit independent models per group with [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/), evaluate with aggregate and per-group metrics, and visualize the results.

## Prerequisites

- Completed [Getting Started](getting-started.md)

<!-- COMPANION_NOTEBOOKS -->

## Load a Panel Dataset

The [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets.fetch_tourism_monthly/) function loads monthly tourism series from the Monash forecasting archive. The full dataset contains 366 series of varying length; we select three long series and drop any rows with missing values:

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y = bunch.frame.select(
    ["time", "T187__tourists", "T188__tourists", "T189__tourists"]
).drop_nulls()
print(f"{len(y)} rows, {len(y.columns)} columns")
print(y.columns)
```

```text
333 rows, 4 columns
['time', 'T187__tourists', 'T188__tourists', 'T189__tourists']
```

Notice the column names use the `{group}__{variable}` convention. The text before `__` identifies the panel group (a tourism region); the text after `__` is the variable name. Every group shares the same variable suffix (`tourists`).

```python
print(y.head())
```

```text
shape: (5, 4)
┌─────────────────────┬────────────────┬────────────────┬────────────────┐
│ time                ┆ T187__tourists ┆ T188__tourists ┆ T189__tourists │
│ ---                 ┆ ---            ┆ ---            ┆ ---            │
│ datetime[μs]        ┆ f64            ┆ f64            ┆ f64            │
╞═════════════════════╪════════════════╪════════════════╪════════════════╡
│ 1980-01-01 00:00:00 ┆ 13328.0        ┆ 4696.0         ┆ 1556.0         │
│ 1980-02-01 00:00:00 ┆ 11352.0        ┆ 4284.0         ┆ 2424.0         │
│ 1980-03-01 00:00:00 ┆ 12048.0        ┆ 3600.0         ┆ 2324.0         │
│ 1980-04-01 00:00:00 ┆ 8876.0         ┆ 3517.0         ┆ 2164.0         │
│ 1980-05-01 00:00:00 ┆ 7708.0         ┆ 3700.0         ┆ 2256.0         │
└─────────────────────┴────────────────┴────────────────┴────────────────┘
```

## Inspect the Panel Structure

The [`inspect_panel`](/pages/api/generated/yohou.utils.inspect_panel/) utility separates global columns (without `__`) from panel groups:

```python
from yohou.utils.panel import inspect_panel

global_names, panel_groups = inspect_panel(y)
print(f"Global columns: {global_names}")
print(f"Panel groups: {panel_groups}")
```

```text
Global columns: []
Panel groups: {'T187': ['T187__tourists'], 'T188': ['T188__tourists'], 'T189': ['T189__tourists']}
```

Each key is a group name, and the value lists the full column names for that group. Since this dataset contains only panel columns (no shared features across groups), `global_names` is empty.

## Explore the Data

[`plot_time_series`](/pages/api/generated/yohou.plotting.plot_time_series/) automatically detects panel columns and creates faceted subplots:

```python
from yohou.plotting import plot_time_series

plot_time_series(y, title="Tourism by Region")
```

Each region shows a clear annual seasonal pattern, but at different scales. T187 has the highest visitor counts, T189 the lowest. This is typical of panel data: shared patterns with group-level differences.

## Train/Test Split

Split the data using [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/), keeping the last 12 months for testing:

```python
from yohou.model_selection import train_test_split

forecasting_horizon = 12
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
print(f"Train: {len(y_train)} months, Test: {len(y_test)} months")
```

```text
Train: 321 months, Test: 12 months
```

## 1. Seasonal Baseline

[`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/) clones a forecaster for each panel group and fits them independently. Start with a [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/) baseline that repeats values from one year ago:

```python
from yohou.point import SeasonalNaive
from yohou.compose import LocalPanelForecaster

baseline = LocalPanelForecaster(
    forecaster=SeasonalNaive(seasonality=12),
)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
```

Behind the scenes, `LocalPanelForecaster` detected three groups from the `__` columns, created three independent `SeasonalNaive` clones, and fitted each on its own unprefixed data.

## 2. Predict

Calling `predict` returns a single DataFrame with all groups, using the same `__` column convention:

```python
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_baseline.head())
```

```text
shape: (5, 5)
┌─────────────────────┬─────────────────────┬────────────────┬────────────────┬────────────────┐
│ time                ┆ vintage_time        ┆ T187__tourists ┆ T188__tourists ┆ T189__tourists │
│ ---                 ┆ ---                 ┆ ---            ┆ ---            ┆ ---            │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64            ┆ f64            ┆ f64            │
╞═════════════════════╪═════════════════════╪════════════════╪════════════════╪════════════════╡
│ 2006-10-01 00:00:00 ┆ 2006-09-01 00:00:00 ┆ 22862.0        ┆ 21689.0        ┆ 12788.0        │
│ 2006-11-01 00:00:00 ┆ 2006-09-01 00:00:00 ┆ 21160.0        ┆ 21229.0        ┆ 14283.0        │
│ 2006-12-01 00:00:00 ┆ 2006-09-01 00:00:00 ┆ 42850.0        ┆ 59550.0        ┆ 9725.0         │
│ 2007-01-01 00:00:00 ┆ 2006-09-01 00:00:00 ┆ 34320.0        ┆ 27144.0        ┆ 8970.0         │
│ 2007-02-01 00:00:00 ┆ 2006-09-01 00:00:00 ┆ 28050.0        ┆ 25696.0        ┆ 14476.0        │
└─────────────────────┴─────────────────────┴────────────────┴────────────────┴────────────────┘
```

The predictions preserve the `__` column convention, so all downstream tools (scorers, plots) work seamlessly with panel data.

## 3. Evaluate

Panel scorers such as [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) and [`MeanSquaredError`](/pages/api/generated/yohou.metrics.MeanSquaredError/) aggregate across all groups by default:

```python
from yohou.metrics import MeanAbsoluteError, MeanSquaredError

mae = MeanAbsoluteError()
mse = MeanSquaredError()
mae.fit(y_train)
mse.fit(y_train)

print(f"MAE={mae.score(y_test, y_pred_baseline):.2f}")
print(f"MSE={mse.score(y_test, y_pred_baseline):.2f}")
```

```text
MAE=1717.56
MSE=4733226.28
```

To see how each region performs, create a scorer that keeps groups separate by aggregating all other dimensions:

```python
mae_per_group = MeanAbsoluteError(
    aggregation_method=["stepwise", "vintagewise", "componentwise"],
)
mae_per_group.fit(y_train)
per_group = mae_per_group.score(y_test, y_pred_baseline)
print(per_group)
```

The `aggregation_method` list controls which dimensions to collapse. By aggregating across steps, vintages, and value columns, we isolate each group's error as a single number.

```text
shape: (1, 3)
┌─────────────┬─────────────┬────────────┐
│ T187__mae   ┆ T188__mae   ┆ T189__mae  │
│ ---         ┆ ---         ┆ ---        │
│ f64         ┆ f64         ┆ f64        │
╞═════════════╪═════════════╪════════════╡
│ 2728.166667 ┆ 1539.666667 ┆ 884.833333 │
└─────────────┴─────────────┴────────────┘
```

Per-group scoring reveals which regions the baseline handles well and which need a more sophisticated model. Regions with stronger trend changes will show higher error.

## 4. Visualize

[`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) automatically detects panel data and creates faceted subplots:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    y_pred_baseline,
    y_train=y_train,
    n_history=36,
    title="SeasonalNaive Baseline (all regions)",
)
```

The `n_history=36` parameter shows the last three years of training data for context. Each panel group gets its own subplot, so you can compare how the seasonal baseline tracks each region.

## 5. Try a Stronger Model

The pipeline is model-agnostic. Replace SeasonalNaive with a [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) that uses [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) features wrapped in a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) and a Ridge regressor:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.compose import FeaturePipeline
from yohou.preprocessing import LagTransformer

ridge_forecaster = LocalPanelForecaster(
    forecaster=PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=FeaturePipeline([
            ("lags", LagTransformer(lag=[1, 2, 3, 6, 12])),
        ]),
    ),
)
ridge_forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = ridge_forecaster.predict(forecasting_horizon=forecasting_horizon)
```

Notice that only the inner forecaster changed. `LocalPanelForecaster` still handles cloning, group splitting, and reassembly automatically.

## 6. Compare Models

Pass a dict of predictions to `plot_forecast` to overlay multiple models on the same chart:

```python
predictions = {
    "SeasonalNaive": y_pred_baseline,
    "Ridge + Lags": y_pred_ridge,
}

plot_forecast(
    y_test,
    predictions,
    y_train=y_train,
    n_history=36,
    title="Model Comparison (all regions)",
)
```

For a metric-level comparison across groups, [`plot_group_scores`](/pages/api/generated/yohou.plotting.plot_group_scores/) visualizes per-group scores as a bar chart:

```python
from yohou.plotting import plot_group_scores

plot_group_scores(
    mae,
    y_test,
    predictions,
    title="MAE by Region",
)
```

You should see the Ridge model achieving lower error than SeasonalNaive across regions, with the improvement varying by group.

## What You Built

In this tutorial, we:

- Loaded a multi-series tourism dataset with the `__` panel naming convention using [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets.fetch_tourism_monthly/)
- Inspected panel structure with [`inspect_panel`](/pages/api/generated/yohou.utils.inspect_panel/) and explored it visually with [`plot_time_series`](/pages/api/generated/yohou.plotting.plot_time_series/)
- Fitted independent models per group using [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/), first with a seasonal baseline, then with a Ridge reduction pipeline
- Evaluated with aggregate and per-group metrics using `aggregation_method`
- Visualized panel forecasts with faceted subplots and compared models with [`plot_group_scores`](/pages/api/generated/yohou.plotting.plot_group_scores/)

## Next Steps

- [Work with Panel Data](../how-to/panel-data.md) for extracting groups, pivoting, and advanced panel operations
- [Panel Data](../explanation/panel-data.md) for the conceptual background on the naming convention and panel strategies
