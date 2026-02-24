# Stationarity

### Decomposition ([View](/examples/stationarity/decomposition/) | [Open in marimo](/examples/stationarity/decomposition/edit/))

**Trend and Seasonality Decomposition**

Decompose time series into trend, seasonality, and residual components using `PolynomialTrendForecaster`, `PatternSeasonalityForecaster`, `FourierSeasonalityForecaster`, and `DecompositionPipeline`. The foundation for stationary modeling.

### Fourier Tuning ([View](/examples/stationarity/fourier_tuning/) | [Open in marimo](/examples/stationarity/fourier_tuning/edit/))

**Fourier Seasonality Harmonic Selection**

Tune the number of harmonics and estimator choice (ElasticNet, Ridge) for Fourier seasonality. Compares Fourier against Pattern seasonality to help choose the right approach for your data.

### Stationarity Transforms ([View](/examples/stationarity/stationarity_transforms/) | [Open in marimo](/examples/stationarity/stationarity_transforms/edit/))

**Differencing, Log, Box-Cox, and Beyond**

Apply stationarity transformers: `LogTransformer`, `BoxCoxTransformer`, `SeasonalDifferencing`, `SeasonalLogDifferencing`, `SeasonalReturn`, and `ASinhTransformer`. Each with invertible transforms for back-transforming predictions.

### Panel Stationarity ([View](/examples/stationarity/panel_stationarity/) | [Open in marimo](/examples/stationarity/panel_stationarity/edit/))

**Per-Group Decomposition and Stationarity**

Apply decomposition (trend + seasonality) and stationarity transformations per panel group. Shows how stationarity techniques scale to grouped time series.
