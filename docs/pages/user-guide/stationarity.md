# Stationarity

This chapter covers making time series stationary through decomposition, trend and seasonality estimation, and stationarity transforms. It also covers the `DecompositionPipeline` for composing trend + seasonality + residual forecasters.

!!! info "Under Development"
    This chapter is being written. Section headings show the planned structure.

**API Reference**: [`yohou.stationarity`](../api/stationarity.md) · [`yohou.compose`](../api/compose.md)
**Examples**: [Stationarity](../examples/stationarity.md)

## Stationarity Overview

## Trend Estimation

### PolynomialTrendForecaster

## Seasonality Estimation

### FourierSeasonalityForecaster

### PatternSeasonalityForecaster

## Stationarity Transforms

### SeasonalDifferencing

### SeasonalLogDifferencing

### SeasonalReturn and AbsoluteSeasonalReturn

### LogTransformer

### BoxCoxTransformer

### ASinhTransformer

## Decomposition Pipeline

### DecompositionPipeline

### Combining Trend + Seasonality + Residual

### Inverse Transform and Reconstruction
