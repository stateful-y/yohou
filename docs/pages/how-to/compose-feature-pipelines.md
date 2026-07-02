# How to Compose Feature Pipelines

This guide shows you how to combine transformers into feature engineering pipelines using [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/), [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/), and [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/). Use this when a single transformer is not enough and you need sequential processing, parallel feature branches, or both.

## Prerequisites

- Familiarity with transformers ([How to Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Understanding of feature pipelines ([Feature Pipelines](../explanation/feature-pipelines.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Chain Transformers Sequentially

[`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) chains transformers so that the output of each step feeds into the next. Pass a list of `(name, transformer)` tuples:

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

pipeline = FeaturePipeline([
    ("diff", SeasonalDifferencing(seasonality=12)),
    ("lags", LagTransformer(lag=[1, 2, 3])),
])

pipeline.fit(y_train)
y_transformed = pipeline.transform(y_train)
```

Steps execute in order: [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) removes the seasonal component, then [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) creates autoregressive features from the differenced series.

The pipeline's `observation_horizon` is the cumulative sum across all steps, since each step's output feeds into the next:

```python
print(pipeline.observation_horizon)  # 12 + 3 = 15
```

## Run Transformers in Parallel

[`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) runs multiple transformers on the same input and concatenates their outputs column-wise:

```python
from yohou.compose import FeatureUnion
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

features = FeatureUnion([
    ("lags", LagTransformer(lag=[1, 3, 6, 12])),
    ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
])

features.fit(y_train)
y_features = features.transform(y_train)
```

Since all branches receive the same input, the union's `observation_horizon` is the maximum across its transformers (not the sum):

```python
print(features.observation_horizon)  # max(12, 12) = 12
```

## Apply Different Transformers to Different Columns

[`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) routes each column subset to a dedicated transformer, then concatenates the results. Use this instead of `FeatureUnion` when different columns need different treatment:

```python
from yohou.compose import ColumnTransformer
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

ct = ColumnTransformer(
    transformers=[
        ("lags", LagTransformer(lag=[1, 2, 3]), ["temperature"]),
        ("rolling", RollingStatisticsTransformer(window_size=7), ["humidity"]),
    ],
    remainder="drop",
)

ct.fit(y_train)
y_features = ct.transform(y_train)
```

Set `remainder="passthrough"` to keep columns not assigned to any transformer. Set it to a transformer instance to apply a default transformation to unmatched columns:

```python
ct = ColumnTransformer(
    transformers=[
        ("rolling", RollingStatisticsTransformer(window_size=7), ["humidity"]),
    ],
    remainder=LagTransformer(lag=[1, 2, 3]),  # default for all other columns
)
```

Like `FeatureUnion`, the `observation_horizon` is the maximum across all column transformers (including the remainder).

## Nest Sequential and Parallel Stages

Place a `FeatureUnion` or `ColumnTransformer` inside a `FeaturePipeline` to first apply a shared preprocessing step, then branch into parallel feature extractors:

```python
from yohou.compose import FeaturePipeline, FeatureUnion
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer
from yohou.stationarity import SeasonalDifferencing

feature_transformer = FeaturePipeline([
    ("diff", SeasonalDifferencing(seasonality=12)),
    ("features", FeatureUnion([
        ("lags", LagTransformer(lag=[1, 3, 6, 12])),
        ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
    ])),
])
```

The same pattern works with `ColumnTransformer` when features need column-specific treatment:

```python
from yohou.compose import FeaturePipeline, ColumnTransformer
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

feature_transformer = FeaturePipeline([
    ("features", ColumnTransformer(
        transformers=[
            ("lags", LagTransformer(lag=[1, 3, 6]), ["temperature"]),
            ("rolling", RollingStatisticsTransformer(window_size=12), ["humidity"]),
        ],
        remainder="passthrough",
    )),
])
```

Pass the composed transformer to a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/):

```python
from yohou.point import PointReductionForecaster
from sklearn.linear_model import Ridge

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=feature_transformer,
)
forecaster.fit(y_train, forecasting_horizon=12)
predictions = forecaster.predict()
```

## Access Named Steps

Use `named_steps` on a `FeaturePipeline` and `named_transformers` on a `FeatureUnion` to inspect or retrieve individual components after construction:

```python
pipeline = FeaturePipeline([
    ("diff", SeasonalDifferencing(seasonality=12)),
    ("lags", LagTransformer(lag=[1, 3, 6, 12])),
])

pipeline.named_steps["diff"]         # SeasonalDifferencing(seasonality=12)
pipeline.named_steps["lags"].lag     # [1, 3, 6, 12]
```

Bracket indexing also works by position or name:

```python
pipeline[0]       # first step
pipeline["diff"]  # same as named_steps["diff"]
pipeline[0:1]     # slice returns a new FeaturePipeline
```

## Tune Nested Parameters

Both `FeaturePipeline` and `FeatureUnion` support `get_params` / `set_params` with double-underscore notation for nested access:

```python
feature_transformer.set_params(features__lags__lag=[1, 2, 3])
feature_transformer.get_params()["features__lags__lag"]  # [1, 2, 3]
```

This integrates with hyperparameter search. See [How to Tune Hyperparameters](tune-hyperparameters.md) for details.

## Use Pipelines with Panel Data

Pipelines work with panel data automatically. When column names use the `__` naming convention, each transformer applies independently to each group's columns. No special configuration is needed:

```python
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=feature_transformer,
)
# Transformers apply per group automatically
forecaster.fit(y_panel, forecasting_horizon=12)
```

For background on panel data conventions and setup, see [How to Work with Panel Data](panel-data.md).

## See Also

- [How to Use Preprocessing Transformers](use-preprocessing-transformers.md) for individual transformer patterns
- [How to Tune Hyperparameters](tune-hyperparameters.md) for searching over pipeline parameters
- [Feature Pipelines](../explanation/feature-pipelines.md) for the conceptual architecture
