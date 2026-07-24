---
template: api-submodule.html
---

# yohou.preprocessing

Preprocessing transformers for stationarization and feature engineering.

### Classes

| Name | Description |
|------|-------------|
| [`ArithmeticTransformer`](generated/yohou.preprocessing.ArithmeticTransformer.md) | Invertible column-wise arithmetic between two columns. |
| [`ReduceTransformer`](generated/yohou.preprocessing.ReduceTransformer.md) | Invertible n-ary reduction of several columns with ``sum`` or ``product``. |
| [`CalendarFeatureTransformer`](generated/yohou.preprocessing.CalendarFeatureTransformer.md) | Extract calendar-based features from the time column. |
| [`HolidayFeatureTransformer`](generated/yohou.preprocessing.HolidayFeatureTransformer.md) | Extract holiday indicator features from a user-provided holiday calendar. |
| [`FunctionTransformer`](generated/yohou.preprocessing.FunctionTransformer.md) | Constructs a transformer from an arbitrary callable. |
| [`SeasonalImputer`](generated/yohou.preprocessing.SeasonalImputer.md) | Seasonal decomposition-based imputation for missing values. |
| [`SimpleImputer`](generated/yohou.preprocessing.SimpleImputer.md) | Simple imputation using sklearn's SimpleImputer. |
| [`SimpleTimeImputer`](generated/yohou.preprocessing.SimpleTimeImputer.md) | Time series imputation using interpolation or filling methods. |
| [`TransformedSpaceKNNImputer`](generated/yohou.preprocessing.TransformedSpaceKNNImputer.md) | K-nearest neighbors imputation in a transformed feature space. |
| [`OutlierPercentileHandler`](generated/yohou.preprocessing.OutlierPercentileHandler.md) | Handle outliers based on percentile thresholds. |
| [`OutlierThresholdHandler`](generated/yohou.preprocessing.OutlierThresholdHandler.md) | Handle outliers based on fixed threshold values. |
| [`Downsampler`](generated/yohou.preprocessing.Downsampler.md) | Downsample time series to a lower frequency using aggregation. |
| [`Upsampler`](generated/yohou.preprocessing.Upsampler.md) | Upsample time series to a higher frequency using interpolation. |
| [`NumericalDifferentiator`](generated/yohou.preprocessing.NumericalDifferentiator.md) | Numerical differentiation transformer for time series signals. |
| [`NumericalFilter`](generated/yohou.preprocessing.NumericalFilter.md) | Apply digital IIR or FIR filters to time series data. |
| [`NumericalIntegrator`](generated/yohou.preprocessing.NumericalIntegrator.md) | Numerical integration transformer for time series signals. |
| [`SklearnScaler`](generated/yohou.preprocessing.SklearnScaler.md) | Wrapper to integrate sklearn scalers into the Yohou pipeline. |
| [`SklearnTransformer`](generated/yohou.preprocessing.SklearnTransformer.md) | Wrapper to integrate sklearn transformers into the Yohou pipeline. |
| [`MaxAbsScaler`](generated/yohou.preprocessing.MaxAbsScaler.md) | Scale each feature by its maximum absolute value. |
| [`MinMaxScaler`](generated/yohou.preprocessing.MinMaxScaler.md) | Transform features by scaling each feature to a given range. |
| [`Normalizer`](generated/yohou.preprocessing.Normalizer.md) | Normalize samples individually to unit norm. |
| [`PolynomialFeatures`](generated/yohou.preprocessing.PolynomialFeatures.md) | Generate polynomial and interaction features. |
| [`PowerTransformer`](generated/yohou.preprocessing.PowerTransformer.md) | Apply a power transform featurewise to make data more Gaussian-like. |
| [`QuantileTransformer`](generated/yohou.preprocessing.QuantileTransformer.md) | Transform features using quantiles information. |
| [`RobustScaler`](generated/yohou.preprocessing.RobustScaler.md) | Scale features using statistics that are robust to outliers. |
| [`SplineTransformer`](generated/yohou.preprocessing.SplineTransformer.md) | Generate univariate B-spline bases for features. |
| [`StandardScaler`](generated/yohou.preprocessing.StandardScaler.md) | Standardize features by removing the mean and scaling to unit variance. |
| [`FourierFeatureTransformer`](generated/yohou.preprocessing.FourierFeatureTransformer.md) | Generate Fourier harmonic features from the time column. |
| [`TimeIndexTransformer`](generated/yohou.preprocessing.TimeIndexTransformer.md) | Convert the time column to a numeric index with optional polynomial terms. |
| [`ExponentialMovingAverage`](generated/yohou.preprocessing.ExponentialMovingAverage.md) | Exponentially Weighted Moving Average (EWMA) transformer. |
| [`LagTransformer`](generated/yohou.preprocessing.LagTransformer.md) | Create lagged features from time series data. |
| [`MeanLagTransformer`](generated/yohou.preprocessing.MeanLagTransformer.md) | Create mean-lagged features by averaging across lag multiples. |
| [`RollingStatisticsTransformer`](generated/yohou.preprocessing.RollingStatisticsTransformer.md) | Compute rolling window statistics for time series. |
| [`SlidingWindowFunctionTransformer`](generated/yohou.preprocessing.SlidingWindowFunctionTransformer.md) | Transform time series by applying a function over sliding windows. |
