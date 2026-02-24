# Preprocessing

### Data Cleaning ([View](/examples/preprocessing/data_cleaning/) | [Open in marimo](/examples/preprocessing/data_cleaning/edit/))

**Imputation and Outlier Handling**

Handle missing values and outliers with `SimpleTimeImputer`, `OutlierThresholdHandler`, and `OutlierPercentileHandler`. Includes `plot_missing_data` for visual gap analysis.

### Advanced Imputation ([View](/examples/preprocessing/advanced_imputation/) | [Open in marimo](/examples/preprocessing/advanced_imputation/edit/))

**Seasonal Imputation and sklearn Wrappers**

Go beyond simple imputation with `SeasonalImputer` and sklearn transformer wrappers. Tests strategies on synthetic gaps with known ground truth.

### Resampling ([View](/examples/preprocessing/resampling/) | [Open in marimo](/examples/preprocessing/resampling/edit/))

**Changing Time Series Frequency**

Upsample and downsample time series with `Upsampler` and `Downsampler`. Essential for aligning series with different frequencies.

### Advanced Resampling ([View](/examples/preprocessing/resampling_advanced/) | [Open in marimo](/examples/preprocessing/resampling_advanced/edit/))

**Aggregation Strategies and Interpolation Methods**

Explore different aggregation strategies, boundary handling, and interpolation methods for resampling. Covers the edge cases that matter in production.

### Window Transformers ([View](/examples/preprocessing/window_transformers/) | [Open in marimo](/examples/preprocessing/window_transformers/edit/))

**Rolling Statistics, EMA, and Lag Features**

Generate windowed features with `LagTransformer`, `RollingStatisticsTransformer`, `SlidingWindowFunctionTransformer`, and `ExponentialMovingAverage`. The core feature engineering toolkit for time series.

### Function Transformer ([View](/examples/preprocessing/function_transformer/) | [Open in marimo](/examples/preprocessing/function_transformer/edit/))

**Custom Function-Based Transforms**

Wrap arbitrary polars operations as sklearn-compatible transformers using `FunctionTransformer`. Supports both stateful and stateless transforms. The escape hatch when built-in transformers don't fit.

### Signal Processing ([View](/examples/preprocessing/signal_processing/) | [Open in marimo](/examples/preprocessing/signal_processing/edit/))

**Butterworth, Chebyshev, and Bessel Filters**

Apply signal processing transformers: `NumericalFilter` (Butterworth, Chebyshev, Bessel), `NumericalDifferentiator`, and `NumericalIntegrator`. For when your time series needs frequency-domain treatment.

### Sklearn Wrappers ([View](/examples/preprocessing/sklearn_wrappers/) | [Open in marimo](/examples/preprocessing/sklearn_wrappers/edit/))

**Sklearn-Compatible Scalers for Polars DataFrames**

Use familiar sklearn transformers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PowerTransformer`, and more) directly on polars DataFrames. Bridges sklearn's preprocessing ecosystem with Yohou's data model.

### Panel Preprocessing ([View](/examples/preprocessing/panel_preprocessing/) | [Open in marimo](/examples/preprocessing/panel_preprocessing/edit/))

**Panel-Aware Transformations**

Apply transformations that automatically respect panel structure. Includes manual per-group extraction with `get_group_df` and `dict_to_panel`.
