# Overview

An overview of Yohou's capabilities organized by module, with key entry points and links to examples.

## Forecasting

Point and interval forecasting with sklearn-compatible estimators. Supports naive baselines, reduction-based forecasters, and conformal prediction.

```python
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.interval import SplitConformalForecaster
```

**Modules**: [`yohou.point`](../api/point.md) · [`yohou.interval`](../api/interval.md)
**Examples**: [Naive forecasters](../examples/index.md#point-forecasting) · [Reduction forecaster](../examples/index.md#point-forecasting) · [Conformal intervals](../examples/index.md#interval-forecasting)

## Preprocessing

Stateful and stateless time series transformers including lag/rolling/EMA windows, signal processing, sklearn scaler wrappers, imputation, and outlier handling.

```python
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer, StandardScaler
```

**Module**: [`yohou.preprocessing`](../api/preprocessing.md)
**Examples**: [Window transformers](../examples/index.md#preprocessing) · [Sklearn wrappers](../examples/index.md#preprocessing)

## Stationarity

Trend and seasonality estimation, stationarity transforms (differencing, log, BoxCox), and Fourier seasonality.

```python
from yohou.stationarity import SeasonalDifferencing, LogTransformer, PolynomialTrendForecaster
```

**Module**: [`yohou.stationarity`](../api/stationarity.md)
**Examples**: [Decomposition](../examples/index.md#stationarity) · [Stationarity transforms](../examples/index.md#stationarity)

## Composition

Combine forecasters and transformers into pipelines, feature unions, column transformers, and multi-step decomposition workflows.

```python
from yohou.compose import DecompositionPipeline, FeaturePipeline, ColumnForecaster
```

**Module**: [`yohou.compose`](../api/compose.md)
**Examples**: [Pipeline composition](../examples/index.md#composition) · [Decomposition variations](../examples/index.md#composition)

## Class-Probability Forecasting

Forecast categorical time series with calibrated probability distributions. Wraps sklearn classifiers via the reduction pattern.

```python
from yohou.class_proba import ClassProbaReductionForecaster
```

**Module**: [`yohou.class_proba`](../api/class_proba.md)
**Examples**: [Class-probability examples](../examples/class_proba.md)

## Ensemble Forecasting

Combine multiple forecasters using voting. Supports point (mean/median), interval (mean/median/envelope), and class-probability (soft/hard voting) ensembles.

```python
from yohou.ensemble import VotingPointForecaster, VotingIntervalForecaster
```

**Module**: [`yohou.ensemble`](../api/ensemble.md)
**Examples**: [Ensemble examples](../examples/ensemble.md)

## Metrics

Point and interval scoring with per-timestep, per-component, and per-panel aggregation. Conformity scorers for conformal prediction calibration.

```python
from yohou.metrics import MeanAbsoluteError, IntervalScore
scorer = MeanAbsoluteError()
scorer.fit(y_train)
scorer.score(y_test, y_pred)
```

**Module**: [`yohou.metrics`](../api/metrics.md)
**Examples**: [Point metrics](../examples/index.md#metrics) · [Interval metrics](../examples/index.md#metrics)

## Model Selection

Time series cross-validation splitters (expanding window, sliding window) and hyperparameter search (grid, randomized) with no data leakage across time.

```python
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
```

**Module**: [`yohou.model_selection`](../api/model_selection.md)
**Examples**: [CV splitters](../examples/index.md#model-selection) · [Hyperparameter search](../examples/index.md#model-selection)

## Visualization

Plotly-based plotting functions for exploration, diagnostics, forecasting, evaluation, and model selection, all with panel data support.

```python
from yohou.plotting import plot_forecast, plot_time_series, plot_autocorrelation
```

**Module**: [`yohou.plotting`](../api/plotting.md)
**Examples**: [Exploration](../examples/index.md#plotting) · [Forecasting visualization](../examples/index.md#plotting)

## Panel Data

First-class support for multiple related time series using the `__` column naming convention. Panel-aware forecasting, scoring, and visualization.

```python
from yohou.utils.panel import inspect_panel
global_cols, groups = inspect_panel(y)  # Discover panel structure
```

**Utilities**: [`yohou.utils`](../api/utils.md)
**Examples**: [Panel forecasting](../examples/index.md#point-forecasting) · [Panel preprocessing](../examples/index.md#preprocessing)

## Datasets

10 remote dataset fetchers downloading Monash/Zenodo time series on demand with local Parquet caching. Covers univariate, multivariate, panel, and classification data:

`fetch_tourism_monthly` · `fetch_sunspot` · `fetch_tourism_quarterly` · `fetch_electricity_demand` · `fetch_dominick` · `fetch_pedestrian_counts` · `fetch_hospital` · `fetch_kdd_cup` · `fetch_air_quality_classification` · `fetch_demand_classification`

```python
from yohou.datasets import fetch_tourism_monthly
bunch = fetch_tourism_monthly()
y = bunch.frame  # polars DataFrame with "time" column
```

**Module**: [`yohou.datasets`](../api/datasets.md)
**Examples**: [Dataset explorers](../examples/index.md#datasets)
