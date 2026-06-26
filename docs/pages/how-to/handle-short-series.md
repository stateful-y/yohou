# How to Handle Short Series

This guide shows you how to handle series that are too short to estimate seasonal
patterns, run temporal cross-validation, or calibrate conformal intervals.

## Prerequisites

- Familiarity with cross-validation and model selection
  ([Model Selection](../explanation/model-selection.md))
- Familiarity with panel data concepts ([Work with Panel Data](panel-data.md))
  if you have multiple series

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Choose a Fallback

| Limitation | Recommended fallback |
|---|---|
| Fewer than 2-3 seasonal cycles | Fourier terms instead of pattern decomposition |
| Too few observations for multi-split CV | Minimal two-split evaluation |
| Individual series short, many series available | Global panel pooling |
| Training period shorter than one cycle | Scale-independent metrics (MAPE, sMAPE) |

!!! info "When is a series 'short'?"
    Shortness is relative to the task: seasonal estimation needs 2-3 complete cycles, temporal CV needs roughly 5x the forecast horizon, and scaled metrics (MASE, RMSSE) need at least one full seasonal cycle for the baseline denominator.

## Use Fourier Terms for Seasonality

When you have fewer than 2-3 complete seasonal cycles,
[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/)
is more reliable than pattern-based decomposition. Fourier terms fit smooth
harmonic functions that need less data to converge than estimating a full
seasonal shape:

```python
from yohou.stationarity.seasonality import FourierSeasonalityForecaster

forecaster = FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2])
```

Start with a low `harmonics` order (1 or 2) to avoid overfitting on limited
data.

## Reduce Cross-Validation Splits

When the series is too short for multi-split CV, use the minimum split count.
[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) requires at least two splits. This gives up the
variance estimate that more splits provide but avoids creating splits where the
training set is too small:

```python
from yohou.model_selection import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(n_splits=2, test_size=6)
```

If even two splits exhaust the training data, reduce `test_size` or evaluate on
the final holdout only by slicing the data manually.

## Pool Short Series with panel_strategy="global"

In panel data scenarios where individual series are short but the aggregate
dataset is large, pooling information across groups can compensate. The global
panel strategy trains a single regressor on all series simultaneously:

```python
from yohou.point import PointReductionForecaster
from sklearn.ensemble import GradientBoostingRegressor

forecaster = PointReductionForecaster(
    estimator=GradientBoostingRegressor(),
    panel_strategy="global",
)
```

The global strategy is appropriate when series share a common underlying
pattern. If series are structurally different, the shared regressor may
perform worse than individual models. See
[Work with Panel Data](panel-data.md) for the other panel strategies.

## Switch to Scale-Independent Metrics

MASE and RMSSE scale errors by the in-sample naive forecast error. If the
training period is shorter than one seasonal cycle, the denominator cannot be
computed. Use
[`MeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.point.MeanAbsolutePercentageError/)
(when zero values are not present) or
[`SymmetricMeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.point.SymmetricMeanAbsolutePercentageError/)
as alternatives:

```python
from yohou.metrics import MeanAbsolutePercentageError

scorer = MeanAbsolutePercentageError()
```

When zero values may be present in the target, prefer sMAPE, which keeps the
denominator bounded:

```python
from yohou.metrics import SymmetricMeanAbsolutePercentageError

scorer = SymmetricMeanAbsolutePercentageError()
```

## See Also

- [Handle Long Series](handle-long-series.md) for the opposite problem:
  when history is so large that computation or stale data becomes an issue.
- [Model Selection](../explanation/model-selection.md) for the theory of
  temporal cross-validation and why standard k-fold is invalid for time series.
- [Handle Complex Seasonality](handle-complex-seasonality.md) for
  multi-period seasonal patterns.
