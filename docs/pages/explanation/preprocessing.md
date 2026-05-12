# Preprocessing

Preprocessing in yohou is built around a single abstraction: [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/). Every transformer, whether it computes lag features, scales values, or applies a digital filter, extends this class and follows the same contract. The contract is simple: accept a polars DataFrame with a `"time"` column, return a polars DataFrame with a `"time"` column. What makes yohou's preprocessing distinct from sklearn's is the addition of *temporal state*: the ability for transformers to remember past observations and use them when new data arrives.

## Stateful and Stateless Transformers

Transformers fall into two categories based on whether they need historical context to produce output.

**Stateless transformers** operate on each row independently. Scaling a column by its mean and standard deviation, applying a log transform, or selecting a subset of columns are all stateless operations. The transformer learns parameters during `fit` (the mean and standard deviation, for instance), but once fitted, it can transform any input without needing to know what came before. A [`StandardScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.StandardScaler/) is a typical stateless transformer. It stores the fitted statistics, but each row's transformation depends only on that row's values and the stored statistics.

**Stateful transformers** need a lookback window of past data to compute their output. A [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) with `lag=3` needs 3 previous rows to produce a valid lagged value for the current row. A [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/) computing a 7-day rolling mean needs 7 days of history. Without that history, the first rows of output would be incomplete, and yohou handles this by dropping them rather than filling with nulls.

The distinction matters most during *streaming* or *rolling evaluation* scenarios, where new observations arrive incrementally and you need to transform them without refitting the entire model.

## The Observation Horizon

The `observation_horizon` property is what makes the stateful/stateless distinction
concrete (see also [Core Concepts](core-concepts.md#observation-horizon) for how
forecasters compose observation horizons across their transformers). It declares how
many past rows a transformer requires to produce valid output. Stateless transformers have `observation_horizon == 0`. Stateful transformers set it to whatever their lookback requires: for a `LagTransformer(lag=[1, 3])`, it is 3 (the maximum lag).

This property shapes behavior across the transformer's lifecycle:

**During `fit`**, the transformer stores the last `observation_horizon` rows in an internal memory buffer (`_X_observed`). These rows become the lookback context for future incremental transforms.

**During `transform`**, the operation is stateless: the transformer treats the input as a self-contained dataset. For stateful transformers, this means the first `observation_horizon` rows are dropped from the output because they lack sufficient history. A 100-row input through a transformer with `observation_horizon=3` produces 97 rows.

**During `observe_transform`**, the transformer concatenates its stored memory with the new input before transforming, then updates the memory buffer. This is the key method for streaming scenarios. Because the memory provides the lookback context, all input rows produce valid output; nothing is dropped. After transformation, `observe` updates the memory with the new data.

**During `rewind_transform`**, the transformer performs a stateless transform (dropping the first `observation_horizon` rows) and then rewinds its internal memory to the end of the input. This is useful when you want to reset the transformer's state to a particular point in time without using pre-existing memory.

The `observe` and `rewind` methods manage the memory buffer directly. `observe` appends new data to the buffer and then calls `rewind` to trim it back to exactly `observation_horizon` rows. `rewind` sets the buffer to the last `observation_horizon` rows of whatever data you provide. Together they maintain a sliding window of recent history.

## Composing Transformers

Real-world feature engineering rarely involves a single transformation. Yohou provides
three composition patterns, each mirroring an sklearn counterpart but adapted for
time series:

- [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/)
chains transformers sequentially. Its combined `observation_horizon` is the **sum**
across all steps because each step's output (minus its lookback overhead) feeds into
the next.

- [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)
runs transformers in parallel on the same input and concatenates outputs column-wise.
Its combined `observation_horizon` is the **maximum** across all transformers.

- [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/)
applies different transformers to different column subsets, then concatenates the
results.

These composites are commonly used as the `target_transformer` or
`feature_transformer` parameter in forecasters. See [Composition](composition.md)
for a deeper discussion of how these patterns interact with observe/rewind state
propagation.

## Bridging sklearn with Polars

Sklearn's extensive library of transformers operates on NumPy arrays and expects no `"time"` column. [`SklearnTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_base.SklearnTransformer/) and [`SklearnScaler`](/pages/api/generated/yohou.preprocessing.sklearn_base.SklearnScaler/) bridge this gap by wrapping any sklearn-compatible transformer to work with polars DataFrames. They handle the conversion automatically: strip the `"time"` column, convert to NumPy, apply the sklearn transformer, convert back to polars, and reattach the `"time"` column.

Pre-built wrappers are provided for the most common sklearn transformers: [`StandardScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.StandardScaler/), [`MinMaxScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.MinMaxScaler/), [`RobustScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.RobustScaler/), [`MaxAbsScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.MaxAbsScaler/), [`Normalizer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.Normalizer/), [`PolynomialFeatures`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.PolynomialFeatures/), [`PowerTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.PowerTransformer/), [`QuantileTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.QuantileTransformer/), and [`SplineTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.SplineTransformer/). These are thin subclasses that set the correct default sklearn class, so you can use them directly without specifying the `transformer` parameter.

For any other sklearn transformer, you can wrap it on the fly:

```python
from sklearn.preprocessing import KBinsDiscretizer
from yohou.preprocessing import SklearnTransformer

discretizer = SklearnTransformer(transformer=KBinsDiscretizer, n_bins=5, strategy="uniform")
```

All wrapped transformers remain stateless (`observation_horizon == 0`) since sklearn transformers have no concept of temporal lookback.

## Key Transformer Categories

Beyond scaling and sklearn wrappers, yohou provides transformers for common time series feature engineering tasks:

**Window transformers** create features from temporal neighborhoods. [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) shifts columns by specified time steps, [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/) computes rolling aggregates (mean, std, min, max, and more), [`ExponentialMovingAverage`](/pages/api/generated/yohou.preprocessing.window.ExponentialMovingAverage/) applies exponential smoothing, and [`SlidingWindowFunctionTransformer`](/pages/api/generated/yohou.preprocessing.window.SlidingWindowFunctionTransformer/) lets you apply any custom function over a sliding window. All of these are stateful: they set `observation_horizon` to their window size.

**Function transformers** apply arbitrary operations. [`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/) takes a callable and applies it element-wise or column-wise, similar to sklearn's version but for polars DataFrames with time column preservation. It supports both forward and inverse transforms.

**Signal processing transformers** apply digital filtering and calculus operations. [`NumericalFilter`](/pages/api/generated/yohou.preprocessing.signal.NumericalFilter/) supports IIR and FIR filters (Butterworth, Chebyshev, Bessel) for noise removal. [`NumericalIntegrator`](/pages/api/generated/yohou.preprocessing.signal.NumericalIntegrator/) and [`NumericalDifferentiator`](/pages/api/generated/yohou.preprocessing.signal.NumericalDifferentiator/) compute numerical integration and differentiation using scipy methods.

**Imputation transformers** handle missing values. [`SimpleImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleImputer/) fills gaps with constant, mean, median, or forward/backward fill strategies. [`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.imputation.SeasonalImputer/) uses seasonal patterns for imputation. [`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/) inserts missing time steps. [`TransformedSpaceKNNImputer`](/pages/api/generated/yohou.preprocessing.imputation.TransformedSpaceKNNImputer/) uses k-nearest neighbors in a transformed feature space.

**Resampling transformers** change time series frequency. [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) aggregates to a lower frequency and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) interpolates to a higher frequency.

## Time Features

Time feature transformers derive exogenous features from the time column itself,
capturing temporal patterns without requiring external data sources.

[`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.CalendarFeatureTransformer/)
extracts integer features from timestamps: month, day of week, hour, and other
calendar components. These features capture regular patterns like weekly or monthly
seasonality in a form that tabular models can use directly.

[`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/)
produces binary indicator features marking public holidays. Holiday effects often
cause sharp deviations from normal patterns (e.g., retail sales spikes, reduced
energy consumption) that seasonal features alone cannot capture.

[`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/)
generates sine and cosine pairs at specified seasonal periods:
$[\sin(2\pi k t / P), \cos(2\pi k t / P)]$ for harmonics $k = 1, \ldots, K$.
Fourier features capture smooth seasonal patterns and are particularly effective
for modeling daily or annual cycles in sub-seasonal data.

[`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/)
converts timestamps to a normalized numeric index, useful for feeding time
information to models that cannot handle datetime types directly.

These transformers are often combined using `FeatureUnion` or `FeaturePipeline`
and passed as the `feature_transformer` parameter to forecasters. See the
[how-to guide on time features](../how-to/time-features.md) for recipes.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapter 13.9 (missing values and outliers).

## Connections

Preprocessing sits between raw data and the forecasting models. Transformers are
passed to forecasters as `target_transformer` or `feature_transformer` parameters,
where they are applied automatically during fit and predict. The
[Stationarity](stationarity.md) transforms (differencing, decomposition) follow the
same `BaseTransformer` contract but focus specifically on making time series
stationary. For how transformers compose inside forecasters and pipelines, see
[Composition](composition.md).

Practical examples: [Data Cleaning](/examples/preprocessing/data_cleaning/),
[Window Transformers](/examples/preprocessing/window_transformers/), and
[Pipeline Composition](/examples/compose/pipeline_composition/).
