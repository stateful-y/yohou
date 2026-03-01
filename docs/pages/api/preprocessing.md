---
template: api-submodule.html
---

# yohou.preprocessing

Time series transformers for feature engineering, scaling, imputation, outlier handling, and resampling.

### Data imputation

| Name | Description |
| --- | --- |
| [`SeasonalImputer`](generated/yohou.preprocessing.imputation.SeasonalImputer.md) | Seasonal decomposition-based imputation for missing values. |
| [`SimpleImputer`](generated/yohou.preprocessing.imputation.SimpleImputer.md) | Simple imputation using sklearn's SimpleImputer. |
| [`SimpleTimeImputer`](generated/yohou.preprocessing.imputation.SimpleTimeImputer.md) | Time series imputation using interpolation or filling methods. |
| [`TransformedSpaceKNNImputer`](generated/yohou.preprocessing.imputation.TransformedSpaceKNNImputer.md) | K-nearest neighbors imputation in a transformed feature space. |

### Handling outliers

| Name | Description |
| --- | --- |
| [`OutlierPercentileHandler`](generated/yohou.preprocessing.outlier.OutlierPercentileHandler.md) | Handle outliers based on percentile thresholds. |
| [`OutlierThresholdHandler`](generated/yohou.preprocessing.outlier.OutlierThresholdHandler.md) | Handle outliers based on fixed threshold values. |

### Resampling

| Name | Description |
| --- | --- |
| [`Downsampler`](generated/yohou.preprocessing.resampling.Downsampler.md) | Downsample time series to a lower frequency using aggregation. |
| [`Upsampler`](generated/yohou.preprocessing.resampling.Upsampler.md) | Upsample time series to a higher frequency using interpolation. |

### Scaling

| Name | Description |
| --- | --- |
| [`SklearnScaler`](generated/yohou.preprocessing.sklearn_base.SklearnScaler.md) | Wrapper to integrate sklearn scalers into the Yohou pipeline. |
| [`StandardScaler`](generated/yohou.preprocessing.sklearn_wrappers.StandardScaler.md) | Standardize features by removing the mean and scaling to unit variance. |
| [`MaxAbsScaler`](generated/yohou.preprocessing.sklearn_wrappers.MaxAbsScaler.md) | Scale each feature by its maximum absolute value. |
| [`MinMaxScaler`](generated/yohou.preprocessing.sklearn_wrappers.MinMaxScaler.md) | Transform features by scaling each feature to a given range. |
| [`Normalizer`](generated/yohou.preprocessing.sklearn_wrappers.Normalizer.md) | Normalize samples individually to unit norm. |
| [`RobustScaler`](generated/yohou.preprocessing.sklearn_wrappers.RobustScaler.md) | Scale features using statistics that are robust to outliers. |

### Feature engineering

| Name | Description |
| --- | --- |
| [`SklearnTransformer`](generated/yohou.preprocessing.sklearn_base.SklearnTransformer.md) | Wrapper to integrate sklearn transformers into the Yohou pipeline. |
| [`LagTransformer`](generated/yohou.preprocessing.window.LagTransformer.md) | Create lagged features from time series data. |
| [`ExponentialMovingAverage`](generated/yohou.preprocessing.window.ExponentialMovingAverage.md) | Exponentially Weighted Moving Average (EWMA) transformer. |
| [`PolynomialFeatures`](generated/yohou.preprocessing.sklearn_wrappers.PolynomialFeatures.md) | Generate polynomial and interaction features. |
| [`PowerTransformer`](generated/yohou.preprocessing.sklearn_wrappers.PowerTransformer.md) | Apply a power transform featurewise to make data more Gaussian-like. |
| [`QuantileTransformer`](generated/yohou.preprocessing.sklearn_wrappers.QuantileTransformer.md) | Transform features using quantiles information. |
| [`SplineTransformer`](generated/yohou.preprocessing.sklearn_wrappers.SplineTransformer.md) | Generate univariate B-spline bases for features. |
| [`FunctionTransformer`](generated/yohou.preprocessing.function.FunctionTransformer.md) | Constructs a transformer from an arbitrary callable. |
| [`SlidingWindowFunctionTransformer`](generated/yohou.preprocessing.window.SlidingWindowFunctionTransformer.md) | Transform time series by applying a function over sliding windows. |
| [`RollingStatisticsTransformer`](generated/yohou.preprocessing.window.RollingStatisticsTransformer.md) | Compute rolling window statistics for time series. |

### Signal processing

| Name | Description |
| --- | --- |
| [`NumericalDifferentiator`](generated/yohou.preprocessing.signal.NumericalDifferentiator.md) | Numerical differentiation transformer for time series signals. |
| [`NumericalFilter`](generated/yohou.preprocessing.signal.NumericalFilter.md) | Apply digital IIR or FIR filters to time series data. |
| [`NumericalIntegrator`](generated/yohou.preprocessing.signal.NumericalIntegrator.md) | Numerical integration transformer for time series signals. |
