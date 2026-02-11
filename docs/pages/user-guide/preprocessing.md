# Preprocessing

This chapter covers time series transformers and the pipeline infrastructure for composing them. Yohou provides windowing, signal processing, sklearn scaler wrappers, imputation, outlier handling, and resampling, all with stateful `update`/`reset` support.

!!! info "Under Development"
    This chapter is being written. Section headings show the planned structure.

**API Reference**: [`yohou.preprocessing`](../api/preprocessing.md) · [`yohou.compose`](../api/compose.md)
**Examples**: [Preprocessing](../examples/preprocessing.md)

## Transformers Overview

### Stateful vs. Stateless

### The update/reset Lifecycle

### Observation Horizon

## Pipelines and Composition

### FeaturePipeline

### FeatureUnion

### ColumnTransformer

## Window Transformers

### RollingStatisticsTransformer

### ExponentialMovingAverage

### LagTransformer

### SlidingWindowFunctionTransformer

## Signal Processing

### NumericalFilter

### NumericalDifferentiator

### NumericalIntegrator

## Sklearn Wrappers

### Scalers

### Other Transformers

## Imputation

### SimpleImputer

### TransformedSpaceKNNImputer

### SimpleTimeImputer

### SeasonalImputer

## Outlier Handling

### OutlierThresholdHandler

### OutlierPercentileHandler

## Resampling

### Downsampler

### Upsampler

## Custom Transformers

### FunctionTransformer
