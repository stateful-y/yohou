# Preprocessing

Preprocessing in yohou is built around [`BaseActualTransformer`](/pages/api/generated/yohou.base.transformer.BaseActualTransformer/), the base for transformers over single-axis data. Whether it computes lag features, scales values, or applies a digital filter, such a transformer follows one contract: accept a polars DataFrame with a `"time"` column, return a polars DataFrame with a `"time"` column. What makes yohou's preprocessing distinct from sklearn's is the addition of *temporal state*: the ability for transformers to remember past observations and use them when new data arrives.

This page is about that single-axis kind, which is what nearly every transformer in the library is. Transformers over the forecast channel carry a second time axis and a correspondingly different contract; see [Transformer Kinds](transformer-kinds.md) for the taxonomy and for why the memory API discussed below belongs to the single-axis kind alone.

## Time Series Data Quality

Standard tabular machine learning treats rows as independent samples. A missing
value in one row says nothing about whether neighbouring rows are also missing,
and an outlier is almost always a data error. Time series data breaks both of
these assumptions, and understanding why changes how you reason about
preprocessing.

### Missingness is Temporal, Not Random

In tabular data, the standard assumption is that values are Missing Completely
At Random (MCAR): the probability of a value being absent is unrelated to any
other variable. In time series, missingness is almost always temporally
structured. Sensor outages produce contiguous blocks of missing data. Reporting
delays create gaps at the end of a series. Market closures cause regular
periodic gaps. The *pattern* of missingness carries diagnostic information: a
block of missing values at a known system downtime is a different situation from
scattered gaps that might indicate intermittent data collection failures. This
is why Yohou provides multiple imputation strategies rather than a single
fill-with-mean approach. Forward fill preserves discontinuities and works well
when the last known value is a reasonable proxy. Seasonal imputation respects
periodic structure. Linear interpolation assumes smooth transitions. The right
choice depends on what the missingness pattern tells you about the data
generating process. For step-by-step guidance, see the
[how-to guide on missing data](../how-to/handle-missing-data.md).

### Outliers May Be Events, Not Errors

In tabular ML, outlier detection and removal is standard practice because
extreme values usually indicate measurement errors or data corruption. In time
series, an extreme value at a specific point in time often represents a genuine
event: a holiday sales spike, a supply chain disruption, an infrastructure
failure, or a weather extreme. Removing these values destroys signal rather
than cleaning noise. The distinction matters for forecasting because the goal
is to predict the future, and events recur. If holiday demand spikes are
removed from training data, the model cannot learn to anticipate them. The
diagnostic question is whether an extreme value correlates with a known
external event. If it does, it belongs in the training data (and possibly
deserves an exogenous feature to help the model learn the association). If it
does not correlate with any identifiable cause, it may warrant replacement.
See the [how-to guide on outliers](../how-to/handle-outliers.md) for detection
and treatment procedures.

### Series Length Constrains Method Choice

Many time series methods have implicit minimum data requirements that
tabular methods do not. Seasonal estimation requires at least two full seasonal
cycles to distinguish seasonal patterns from noise (one cycle is not enough
because there is no way to confirm the pattern repeats). Rolling statistics
transformers need a window of history before they produce valid output. Fourier
features for complex seasonality need enough data to estimate harmonics
reliably. When a series is too short for a chosen method, the result is not an
obvious error but a subtly unreliable model: parameters are estimated from
insufficient evidence, and the model overfits the few cycles it has seen.
The practical implication is that preprocessing and model complexity should
scale with available data length. Short series benefit from simpler
transformers and lower-order models, while long series can support richer
feature engineering. See
[handling short series](../how-to/handle-short-series.md) and
[handling long series](../how-to/handle-long-series.md) for concrete
strategies.

### Resampling Changes Information Content

Changing the frequency of a time series is not a neutral operation. Aggregating
from hourly to daily measurements loses intra-day variance: the difference
between a calm day and a volatile day disappears when both are reduced to a
single daily mean. Interpolating from monthly to weekly data invents
observations that were never measured, and the interpolated values reflect the
interpolation method's assumptions rather than actual system behavior. Both
directions involve a tradeoff between the convenience of a uniform frequency
and the fidelity of the data. Aggregation is generally safer (it discards
real information but does not fabricate it), while interpolation should be
used with caution and awareness that the synthetic data points carry less
information than genuine observations. See the
[how-to guide on cleaning and resampling](../how-to/clean-and-resample.md)
for procedures.

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

The `observe` and `rewind` methods manage the memory buffer directly. `observe` appends new data to the buffer, enforcing temporal continuity (the new data's timestamps must follow directly from the stored memory). It then trims the buffer back to exactly `observation_horizon` rows. `rewind` sets the buffer to the last `observation_horizon` rows of whatever data you provide, with no continuity requirement. Together they maintain a sliding window of recent history.

## Invertibility

Some transformers support `inverse_transform`, which reverses the transformation. This is essential for forecasting pipelines where the model trains on transformed data but predictions must be returned in the original scale.

Stateless invertible transformers (such as [`StandardScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.StandardScaler/) or [`PowerTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.PowerTransformer/)) need only the transformed data to invert. Stateful invertible transformers (such as [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/)) require past observations to reconstruct the original values, which are passed via the `X_p` parameter:

```python
from yohou.stationarity import SeasonalDifferencing

diff = SeasonalDifferencing(seasonality=4)
diff.fit(X)

X_diff = diff.transform(X)  # First 4 rows dropped
X_original = diff.inverse_transform(X_t=X_diff, X_p=past_observations)
```

When transformers are composed inside a [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/), the pipeline handles `inverse_transform` automatically by reversing the steps and passing the necessary context.

Transformers that modify values in existing columns (scaling, differencing, power transforms) are typically invertible, while those that create new derived columns (lags, rolling statistics, calendar features) are not.

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
results. Its `observation_horizon` is the **maximum** across all column-specific
transformers.

These composites are commonly used as the `target_transformer` or
`actual_transformer` parameter in forecasters. See [Feature Pipelines](feature-pipelines.md)
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

Beyond scaling and sklearn wrappers, yohou provides transformers for common time
series preprocessing tasks organized into six families. This grouping
reflects the fact that time series preprocessing involves qualitatively different
operations: some create features from temporal neighborhoods, others handle data
quality issues, and others change the resolution of the data itself. Organizing
them by purpose helps you find the right tool for each preprocessing need.

**Window transformers** create features from temporal neighborhoods: lagged values,
rolling aggregates (mean, standard deviation, min, max), exponential moving averages,
and arbitrary functions applied over a sliding window. These are the primary tools for
expressing "what was the series doing recently?" in a form the regressor can learn from.
All window transformers are stateful, with `observation_horizon` determined by their
lookback requirement: `max(lags)` for lag transformers, `window_size - 1` for rolling
statistics, and `1` for exponential moving averages.

**Function transformers** apply arbitrary operations element-wise or column-wise to
a polars DataFrame, with time column preservation and optional inverse transform
support. [`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/)
automatically detects its `observation_horizon` during fitting by counting how many
leading rows produce null output from the custom function, making it adapt to
arbitrary user logic. They serve as an escape hatch for custom transformations that
do not fit neatly into the other families.

**Signal processing transformers** apply digital filtering (IIR and FIR families)
and numerical calculus (integration, differentiation) to time series columns. They
are useful for smoothing noisy sensor data, extracting trend derivatives, or
pre-conditioning signals before feature extraction.

**Imputation transformers** handle missing values and gaps in the time series.
Different imputation strategies make different assumptions: simple forward-fill
preserves discontinuities, linear interpolation works for smooth series,
seasonal imputation accounts for periodic structure, and nearest-neighbor imputation
in a transformed feature space can capture complex multi-variate patterns.

**Outlier handling transformers** detect and treat extreme values using either fixed
thresholds or data-driven percentile bounds. Values outside the specified range can
be clipped to the boundary or replaced with null for downstream imputation. Both
[`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/)
and [`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierPercentileHandler/)
are stateless.

**Resampling transformers** change time series frequency by aggregating to a lower
frequency or interpolating to a higher one. Resampling before modeling is often
necessary when the available data frequency does not match the required forecast
frequency.

For individual class parameters and usage examples, see the
[API Reference: yohou.preprocessing](/pages/api/preprocessing/).

## Time Features

A separate family of transformers derives exogenous features directly from the time
column, capturing temporal patterns without requiring external data sources. Calendar
features (month, day of week, hour) encode regular cyclic patterns as integer columns.
Holiday indicator features mark public holidays as binary signals, with optional
proximity features (days to next holiday, days since last). Fourier feature
pairs ($[\sin(2\pi k t / P), \cos(2\pi k t / P)]$ for harmonics $k = 1, \ldots, K$)
represent smooth seasonal patterns in a compact, continuous form that avoids the
discontinuities of raw calendar integers. A
[`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/)
converts timestamps to a numeric index with optional polynomial expansion, useful for
capturing long-range trends.

These transformers derive entirely from the timestamps, so they can be applied at
predict time without any additional data. They are usually combined via [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)
and passed as the `actual_transformer` parameter to forecasters. For recipes, see
the [how-to guide on time features](../how-to/time-features.md).

## Panel Data Support

All transformers handle panel (grouped) data through yohou's `__` column naming convention. When input columns follow the pattern `{group}__{variable}` (for example, `store_1__sales`, `store_2__sales`), transformers apply their operations independently to each group while preserving the naming structure. Output columns maintain the separator: a [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) applied to `store_1__sales` produces `store_1__sales_lag_1`.

Composition utilities respect this convention as well. When [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) prefixes output features with transformer names, the prefix is inserted after the group separator (`store_1__lags_sales`) rather than before it, keeping group identity as the leading element.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapter 13.9 (missing values and outliers).

## Connections

Preprocessing sits between raw data and the forecasting models. Transformers are
passed to forecasters as `target_transformer` or `actual_transformer` parameters,
where they are applied automatically during fit and predict. The
[Stationarity](stationarity.md) transforms (differencing, decomposition) follow the
same [`BaseActualTransformer`](/pages/api/generated/yohou.base.transformer.BaseActualTransformer/) contract but focus specifically on making time series
stationary. For how transformers compose inside forecasters and pipelines, see
[Feature Pipelines](feature-pipelines.md).

For practical recipes, see [How to Use Preprocessing Transformers](../how-to/use-preprocessing-transformers.md) and [How to Compose Feature Pipelines](../how-to/compose-feature-pipelines.md).
